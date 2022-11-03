<template>
  <div class="text-center">
    <v-dialog v-model="show">
      <v-card style="padding: 10px">
        <img :src="'data:image/png;base64, ' + imgs_bytes[img_index]"/>
        <v-card-text>
          <v-container>
            <v-row>
              <v-col md="12" class="d-flex justify-center">
                {{ img_index + 1 }} of {{ imgs_bytes.length}}
              </v-col>
            </v-row>
            <v-row justify="center" v-if="imgs_bytes.length > 1">
              <v-col md="2" class="d-flex justify-center">
                <v-btn
                  variant="outlined"
                  icon
                  color="primary"
                  @click="prevImage"
                >
                  <v-icon>mdi-chevron-left</v-icon>
                </v-btn>
              </v-col>
              <v-col md="2" class="d-flex justify-center">
                <v-btn
                  variant="outlined"
                  icon
                  color="primary"
                  @click="nextImage"
                >
                  <v-icon>mdi-chevron-right</v-icon>
                </v-btn>
              </v-col>
            </v-row>
          </v-container>
        </v-card-text>
        <v-card-actions>
          <v-btn color="primary" @click="save">Save this image</v-btn>
          <v-spacer></v-spacer>
          <v-btn color="primary" @click="closeModal">Close Dialog</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
import axios from 'axios'
export default {
    props: ['showModal', 'imgs_bytes', 'zs'],
    data() {
        return {
            show: false,
            img_index: 0
        }
    },
    methods: {
        closeModal(){
            this.img_index = 0
            this.$emit('close')
        },
        nextImage(){
            this.img_index = (this.img_index + 1) % this.imgs_bytes.length
        },
        prevImage(){
          this.img_index = (this.img_index + this.imgs_bytes.length - 1) % this.imgs_bytes.length
        },
        save(){
            let formData = new FormData();
            formData.append('z', this.zs[this.img_index])
            axios({
                url: 'https://pf-2021-generadorcaras.it.itba.edu.ar/api/save',
                method: 'post',
                data: formData,
                headers: { "Content-Type": "multipart/form-data" },
            }).then((response) => {
                console.log(response)
                alert('Image saved with ID: ' + response.data.id)
            }).catch((e) => {
                alert('Error')
                console.log(e)
            })
            
        },
    },
    watch: {
        showModal(newValue, oldValue) {
            this.show = newValue
        }
    }
}
</script>

<style scoped>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 40%;
}
</style>